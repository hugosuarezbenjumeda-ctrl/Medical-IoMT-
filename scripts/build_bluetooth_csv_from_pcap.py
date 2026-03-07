#!/usr/bin/env python3
"""Generate CICIoMT-style Bluetooth CSV features from Bluetooth PCAP files.

Methodology implemented from CICIoMT/CICIoT descriptions:
- Parse packet captures with DPKT.
- Compute per-packet flow features.
- Apply fixed-size, non-overlapping window averaging.
- Emit the 45-column schema used by the existing WiFi/MQTT CSVs.

Default Bluetooth window size is 10.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CSV_COLUMNS: List[str] = [
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

EPS = 1e-9

HTTP_PORTS = {80, 8080}
HTTPS_PORTS = {443, 8443}
DNS_PORTS = {53}
TELNET_PORTS = {23}
SMTP_PORTS = {25, 587}
SSH_PORTS = {22}
IRC_PORTS = {194, 6667, 6697}
DHCP_PORTS = {67, 68}


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std_population(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(self.m2 / self.count)

    @property
    def var_population(self) -> float:
        if self.count == 0:
            return 0.0
        return self.m2 / self.count


@dataclass
class RunningCovariance:
    """Online population covariance for paired (x, y) values."""

    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    c: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.count += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.count
        dy = y - self.mean_y
        self.mean_y += dy / self.count
        self.c += dx * (y - self.mean_y)

    @property
    def cov_population(self) -> float:
        if self.count == 0:
            return 0.0
        return self.c / self.count


@dataclass
class FlowState:
    first_ts: float
    prev_ts: Optional[float] = None
    packet_count: int = 0
    src_packets: int = 0
    dst_packets: int = 0
    fin_total: int = 0
    syn_total: int = 0
    rst_total: int = 0
    ack_total: int = 0
    in_stats: RunningStats = field(default_factory=RunningStats)
    out_stats: RunningStats = field(default_factory=RunningStats)
    paired_cov: RunningCovariance = field(default_factory=RunningCovariance)
    pending_in: deque = field(default_factory=deque)
    pending_out: deque = field(default_factory=deque)
    size_stats: RunningStats = field(default_factory=RunningStats)


class WindowAverager:
    def __init__(self, columns: List[str], window_size: int, include_partial: bool) -> None:
        self.columns = columns
        self.window_size = window_size
        self.include_partial = include_partial
        self.count = 0
        self.sums = [0.0 for _ in columns]

    def add(self, row: Dict[str, float]) -> Optional[List[float]]:
        for i, c in enumerate(self.columns):
            self.sums[i] += float(row[c])
        self.count += 1
        if self.count >= self.window_size:
            return self._flush()
        return None

    def finish(self) -> Optional[List[float]]:
        if self.include_partial and self.count > 0:
            return self._flush()
        return None

    def _flush(self) -> List[float]:
        denom = max(self.count, 1)
        out = [v / denom for v in self.sums]
        self.count = 0
        self.sums = [0.0 for _ in self.columns]
        return out


def canonical_flow_key(
    proto: int,
    src_id: str,
    dst_id: str,
    sport: int,
    dport: int,
) -> Tuple[Tuple[object, ...], bool]:
    a = (src_id, int(sport))
    b = (dst_id, int(dport))
    if a <= b:
        return (int(proto), a, b), True
    return (int(proto), b, a), False


def _try_ipv4(dpkt_mod, buf: bytes):
    try:
        ip = dpkt_mod.ip.IP(buf)
        return ip
    except Exception:
        return None


def parse_packet(dpkt_mod, buf: bytes) -> Dict[str, object]:
    # Defaults for malformed or non-IP packets.
    out: Dict[str, object] = {
        "src_id": "unknown_src",
        "dst_id": "unknown_dst",
        "sport": 0,
        "dport": 0,
        "proto": 0,
        "ttl": 0,
        "header_length": 0.0,
        "is_tcp": 0.0,
        "is_udp": 0.0,
        "is_dhcp": 0.0,
        "is_arp": 0.0,
        "is_icmp": 0.0,
        "is_igmp": 0.0,
        "is_ipv": 0.0,
        "is_llc": 0.0,
        "is_http": 0.0,
        "is_https": 0.0,
        "is_dns": 0.0,
        "is_telnet": 0.0,
        "is_smtp": 0.0,
        "is_ssh": 0.0,
        "is_irc": 0.0,
        "fin_flag": 0.0,
        "syn_flag": 0.0,
        "rst_flag": 0.0,
        "psh_flag": 0.0,
        "ack_flag": 0.0,
        "ece_flag": 0.0,
        "cwr_flag": 0.0,
    }

    pkt_size = float(len(buf))

    eth = None
    payload = None
    try:
        eth = dpkt_mod.ethernet.Ethernet(buf)
        payload = eth.data
        out["header_length"] = 14.0 + (4.0 * len(getattr(eth, "vlan_tags", [])))
    except Exception:
        # Try decoding packet directly as raw IPv4.
        payload = _try_ipv4(dpkt_mod, buf)
        out["header_length"] = 0.0

    if payload is not None and isinstance(payload, dpkt_mod.llc.LLC):
        out["is_llc"] = 1.0

    ip_pkt = None
    if payload is not None and isinstance(payload, dpkt_mod.ip.IP):
        ip_pkt = payload
        out["is_ipv"] = 1.0
        out["proto"] = int(ip_pkt.p)
        out["ttl"] = int(getattr(ip_pkt, "ttl", 0) or 0)
        out["src_id"] = str(ip_pkt.src.hex())
        out["dst_id"] = str(ip_pkt.dst.hex())
        out["header_length"] = float(out["header_length"]) + float(ip_pkt.hl * 4)
    elif payload is not None and hasattr(dpkt_mod, "ip6") and isinstance(payload, dpkt_mod.ip6.IP6):
        ip_pkt = payload
        out["is_ipv"] = 1.0
        out["proto"] = int(ip_pkt.nxt)
        out["ttl"] = int(getattr(ip_pkt, "hlim", 0) or 0)
        out["src_id"] = str(ip_pkt.src.hex())
        out["dst_id"] = str(ip_pkt.dst.hex())
        out["header_length"] = float(out["header_length"]) + 40.0
    elif payload is not None and isinstance(payload, dpkt_mod.arp.ARP):
        arp = payload
        out["is_arp"] = 1.0
        out["proto"] = 0
        out["src_id"] = f"arp:{arp.sha.hex()}"
        out["dst_id"] = f"arp:{arp.tha.hex()}"

    if ip_pkt is not None:
        l4 = ip_pkt.data
        proto = int(out["proto"])

        if proto == 6 and isinstance(l4, dpkt_mod.tcp.TCP):
            out["is_tcp"] = 1.0
            out["sport"] = int(l4.sport)
            out["dport"] = int(l4.dport)
            out["header_length"] = float(out["header_length"]) + float((l4.off or 0) * 4)
            flags = int(l4.flags)
            out["fin_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_FIN else 0.0
            out["syn_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_SYN else 0.0
            out["rst_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_RST else 0.0
            out["psh_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_PUSH else 0.0
            out["ack_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_ACK else 0.0
            out["ece_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_ECE else 0.0
            out["cwr_flag"] = 1.0 if flags & dpkt_mod.tcp.TH_CWR else 0.0
        elif proto == 17 and isinstance(l4, dpkt_mod.udp.UDP):
            out["is_udp"] = 1.0
            out["sport"] = int(l4.sport)
            out["dport"] = int(l4.dport)
            out["header_length"] = float(out["header_length"]) + 8.0
        elif proto == 1:
            out["is_icmp"] = 1.0
            out["header_length"] = float(out["header_length"]) + 8.0
        elif proto == 2:
            out["is_igmp"] = 1.0
            out["header_length"] = float(out["header_length"]) + 8.0

    ports = {int(out["sport"]), int(out["dport"])}
    out["is_http"] = 1.0 if ports & HTTP_PORTS else 0.0
    out["is_https"] = 1.0 if ports & HTTPS_PORTS else 0.0
    out["is_dns"] = 1.0 if ports & DNS_PORTS else 0.0
    out["is_telnet"] = 1.0 if ports & TELNET_PORTS else 0.0
    out["is_smtp"] = 1.0 if ports & SMTP_PORTS else 0.0
    out["is_ssh"] = 1.0 if ports & SSH_PORTS else 0.0
    out["is_irc"] = 1.0 if ports & IRC_PORTS else 0.0
    out["is_dhcp"] = 1.0 if ports & DHCP_PORTS else 0.0

    out["packet_size"] = pkt_size
    return out


def build_feature_row(
    state: FlowState,
    ts: float,
    parsed: Dict[str, object],
    is_forward: bool,
) -> Dict[str, float]:
    size = float(parsed["packet_size"])

    state.packet_count += 1
    if is_forward:
        state.src_packets += 1
        state.out_stats.update(size)
        state.pending_out.append(size)
    else:
        state.dst_packets += 1
        state.in_stats.update(size)
        state.pending_in.append(size)

    # Pair incoming/outgoing packet sizes in arrival order for online covariance.
    while state.pending_in and state.pending_out:
        x = float(state.pending_in.popleft())
        y = float(state.pending_out.popleft())
        state.paired_cov.update(x, y)

    if float(parsed["fin_flag"]) > 0:
        state.fin_total += 1
    if float(parsed["syn_flag"]) > 0:
        state.syn_total += 1
    if float(parsed["rst_flag"]) > 0:
        state.rst_total += 1
    if float(parsed["ack_flag"]) > 0:
        state.ack_total += 1

    state.size_stats.update(size)

    elapsed = max(ts - state.first_ts, 0.0)
    rate = (state.packet_count / elapsed) if elapsed > EPS else 0.0
    srate = (state.src_packets / elapsed) if elapsed > EPS else 0.0
    drate = (state.dst_packets / elapsed) if elapsed > EPS else 0.0

    iat = 0.0 if state.prev_ts is None else max(ts - state.prev_ts, 0.0)
    state.prev_ts = ts

    in_var = state.in_stats.var_population
    out_var = state.out_stats.var_population
    mean_in = state.in_stats.mean
    mean_out = state.out_stats.mean
    magnitude = math.sqrt(max(mean_in + mean_out, 0.0))
    radius = math.sqrt(max(in_var + out_var, 0.0))
    covariance = state.paired_cov.cov_population
    variance_ratio = (in_var / out_var) if out_var > EPS else 0.0
    weight = float(state.in_stats.count * state.out_stats.count)

    row = {
        "Header_Length": float(parsed["header_length"]),
        "Protocol Type": float(parsed["proto"]),
        "Duration": float(parsed["ttl"]),
        "Rate": rate,
        "Srate": srate,
        "Drate": drate,
        "fin_flag_number": float(parsed["fin_flag"]),
        "syn_flag_number": float(parsed["syn_flag"]),
        "rst_flag_number": float(parsed["rst_flag"]),
        "psh_flag_number": float(parsed["psh_flag"]),
        "ack_flag_number": float(parsed["ack_flag"]),
        "ece_flag_number": float(parsed["ece_flag"]),
        "cwr_flag_number": float(parsed["cwr_flag"]),
        "ack_count": float(state.ack_total),
        "syn_count": float(state.syn_total),
        "fin_count": float(state.fin_total),
        "rst_count": float(state.rst_total),
        "HTTP": float(parsed["is_http"]),
        "HTTPS": float(parsed["is_https"]),
        "DNS": float(parsed["is_dns"]),
        "Telnet": float(parsed["is_telnet"]),
        "SMTP": float(parsed["is_smtp"]),
        "SSH": float(parsed["is_ssh"]),
        "IRC": float(parsed["is_irc"]),
        "TCP": float(parsed["is_tcp"]),
        "UDP": float(parsed["is_udp"]),
        "DHCP": float(parsed["is_dhcp"]),
        "ARP": float(parsed["is_arp"]),
        "ICMP": float(parsed["is_icmp"]),
        "IGMP": float(parsed["is_igmp"]),
        "IPv": float(parsed["is_ipv"]),
        "LLC": float(parsed["is_llc"]),
        "Tot sum": float(state.size_stats.total),
        "Min": float(state.size_stats.minimum if state.size_stats.count else 0.0),
        "Max": float(state.size_stats.maximum if state.size_stats.count else 0.0),
        "AVG": float(state.size_stats.mean),
        "Std": float(state.size_stats.std_population),
        "Tot size": size,
        "IAT": iat,
        "Number": float(state.packet_count),
        "Magnitue": magnitude,
        "Radius": radius,
        "Covariance": covariance,
        "Variance": variance_ratio,
        "Weight": weight,
    }
    return row


def iter_pcap_packets(dpkt_mod, pcap_path: Path) -> Iterable[Tuple[float, bytes]]:
    with pcap_path.open("rb") as fh:
        # 0x0a0d0d0a marks pcapng.
        magic = fh.read(4)
        fh.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            reader = dpkt_mod.pcapng.Reader(fh)
        else:
            reader = dpkt_mod.pcap.Reader(fh)
        for ts, buf in reader:
            yield float(ts), bytes(buf)


def pcap_to_csv(
    dpkt_mod,
    pcap_path: Path,
    out_path: Path,
    window_size: int,
    include_partial_window: bool,
) -> Tuple[int, int]:
    flows: Dict[Tuple[object, ...], FlowState] = {}
    averager = WindowAverager(CSV_COLUMNS, window_size, include_partial_window)
    rows_in = 0
    rows_out = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(CSV_COLUMNS)

        for ts, buf in iter_pcap_packets(dpkt_mod, pcap_path):
            parsed = parse_packet(dpkt_mod, buf)
            flow_key, is_forward = canonical_flow_key(
                int(parsed["proto"]),
                str(parsed["src_id"]),
                str(parsed["dst_id"]),
                int(parsed["sport"]),
                int(parsed["dport"]),
            )
            state = flows.get(flow_key)
            if state is None:
                state = FlowState(first_ts=ts)
                flows[flow_key] = state

            row = build_feature_row(state, ts, parsed, is_forward)
            rows_in += 1
            maybe = averager.add(row)
            if maybe is not None:
                writer.writerow(maybe)
                rows_out += 1

        tail = averager.finish()
        if tail is not None:
            writer.writerow(tail)
            rows_out += 1

    return rows_in, rows_out


def output_path_for_pcap(root: Path, pcap_path: Path) -> Path:
    rel = pcap_path.relative_to(root)
    parts = list(rel.parts)
    if "pcap" in parts:
        idx = parts.index("pcap")
        parts[idx] = "csv"
    out_dir = root.joinpath(*parts[:-1])
    return out_dir / f"{pcap_path.name}.csv"


def discover_pcap_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.pcap", "*.pcapng"):
        files.extend(root.rglob(ext))
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Bluetooth CICIoMT-style CSV features from PCAP files.",
    )
    parser.add_argument(
        "--input-root",
        default="data/ciciomt2024/Bluetooth",
        help="Bluetooth dataset root containing pcap folders.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Non-overlapping packet window size used for averaging (Bluetooth default: 10).",
    )
    parser.add_argument(
        "--include-partial-window",
        action="store_true",
        help="Also write the last partial window (default: drop partial window).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV outputs (default: skip existing files).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Process only the first N files for smoke tests (0 = all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_size <= 0:
        print("window-size must be > 0", file=sys.stderr)
        return 2

    try:
        import dpkt  # type: ignore
    except Exception:
        print(
            "Missing dependency: dpkt\nInstall with: python3 -m pip install dpkt",
            file=sys.stderr,
        )
        return 2

    root = Path(args.input_root).resolve()
    if not root.exists():
        print(f"Input root does not exist: {root}", file=sys.stderr)
        return 2

    pcap_files = discover_pcap_files(root)
    if args.limit_files > 0:
        pcap_files = pcap_files[: args.limit_files]

    if not pcap_files:
        print(f"No PCAP files found under: {root}", file=sys.stderr)
        return 1

    print(f"Found {len(pcap_files)} PCAP files under {root}")
    print(f"Using window size: {args.window_size}")

    total_packets = 0
    total_rows = 0
    converted = 0
    skipped = 0

    for pcap_path in pcap_files:
        out_path = output_path_for_pcap(root, pcap_path)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            rows_in, rows_out = pcap_to_csv(
                dpkt,
                pcap_path=pcap_path,
                out_path=out_path,
                window_size=args.window_size,
                include_partial_window=args.include_partial_window,
            )
            converted += 1
            total_packets += rows_in
            total_rows += rows_out
            print(f"[ok] {pcap_path} -> {out_path} | packets={rows_in} csv_rows={rows_out}")
        except Exception as exc:
            print(f"[error] {pcap_path}: {exc}", file=sys.stderr)

    print(
        f"Done. converted={converted} skipped={skipped} "
        f"packets={total_packets} csv_rows={total_rows}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
