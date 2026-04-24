"""
tests/test_ack_protocol.py

Pytest suite for the HIAER-Spike hardware ACK protocol.

All hardware DMA I/O is replaced with mocks — no FPGA required.
Tests are organised to mirror the spec sections:

  §2  ACK packet format            → TestParseAckPacket
  §3  Commands that generate ACKs  → TestWriteParametersSimple,
                                     TestWriteNeuronType, TestClear
  §4  C2H packet headers           → TestHeaderConstants
  §5  TX priority / pending ACKs   → TestFlushSpikes (pending-ACK cases)
  §6  Sequence number & status     → TestWaitForAck, TestDrainAcks
  §7  Host software integration    → TestRead, TestReadSelect,
                                     TestFlushSpikes (normal cases)
"""

import sys
import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Mock the compiled Cython DMA extension BEFORE any hs_bridge imports.
# fpga_controller imports `dmadump` at module load time, so sys.modules must
# be patched first.
# ---------------------------------------------------------------------------
_mock_dmadump = MagicMock()
_mock_dmadump.DmaMethodNormal = 0
sys.modules.setdefault("hs_bridge.wrapped_dmadump", MagicMock())
sys.modules.setdefault("hs_bridge.wrapped_dmadump.dmadump", _mock_dmadump)

import hs_bridge.FPGA_Execution.fpga_controller as fc  # noqa: E402
from hs_bridge.FPGA_Execution.fpga_controller import (  # noqa: E402
    ACKError,
    ACK_STATUS_BACKPRESSURE,
    ACK_STATUS_OK,
    _HDR_ACK,
    _HDR_EXEC_DONE,
    _HDR_FIFO_EMPTY,
    _HDR_HBM_ACC,
    _HDR_HBM_DATA,
    _HDR_LATENCY,
    _HDR_MEM_POT,
    _HDR_SPIKE,
    clear,
    drain_acks,
    flush_spikes,
    parse_ack_packet,
    read,
    readSelect,
    wait_for_ack,
    write_neuron_type,
    write_parameters_simple,
)


# ===========================================================================
# Packet builder helpers
# ===========================================================================

def make_ack(cmd_echo: int, seq_num: int = 0, status: int = ACK_STATUS_OK) -> np.ndarray:
    """64-element ACK packet (header 0xACE0_ACE0, spec §2)."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xAC; p[62] = 0xE0; p[61] = 0xAC; p[60] = 0xE0  # header
    p[59] = cmd_echo & 0xFF
    p[58] = (seq_num >> 24) & 0xFF
    p[57] = (seq_num >> 16) & 0xFF
    p[56] = (seq_num >>  8) & 0xFF
    p[55] =  seq_num        & 0xFF
    p[54] = status
    return p


def make_fifo_empty() -> np.ndarray:
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xFF; p[62] = 0xFF
    return p


def make_spike_pkt() -> np.ndarray:
    """0xEEEE_EEEE spike packet with no valid spikes (all-zero payload)."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xEE; p[62] = 0xEE; p[61] = 0xEE; p[60] = 0xEE
    return p


def make_exec_done_pkt() -> np.ndarray:
    """0xABCD_ABCD execution-done packet, no valid spikes in payload."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xAB; p[62] = 0xCD; p[61] = 0xAB; p[60] = 0xCD
    return p


def make_latency_pkt(latency: int = 100) -> np.ndarray:
    """0xBABA_BABA latency counter packet. Value in bits[31:0] (bytes[3:0] LE)."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xBA; p[62] = 0xBA; p[61] = 0xBA; p[60] = 0xBA
    p[0] =  latency        & 0xFF
    p[1] = (latency >>  8) & 0xFF
    p[2] = (latency >> 16) & 0xFF
    p[3] = (latency >> 24) & 0xFF
    return p


def make_hbm_acc_pkt(count: int = 7) -> np.ndarray:
    """0xCABA_CABA HBM access counter packet. Value in bits[31:0] (bytes[3:0] LE)."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xCA; p[62] = 0xBA; p[61] = 0xCA; p[60] = 0xBA
    p[0] =  count        & 0xFF
    p[1] = (count >>  8) & 0xFF
    p[2] = (count >> 16) & 0xFF
    p[3] = (count >> 24) & 0xFF
    return p


def make_mem_pot_pkt() -> np.ndarray:
    """0xCCCC____ membrane potential readback (all-zero payload)."""
    p = np.zeros(64, dtype=np.uint64)
    p[63] = 0xCC; p[62] = 0xCC
    return p


def seq_reads(*packets):
    """
    Build a dma_dump_read side_effect that yields (0, pkt) for each packet
    in order, then returns FIFO-empty once exhausted.
    """
    it = iter(packets)

    def _read(*args, **kwargs):
        try:
            return (0, next(it))
        except StopIteration:
            return (0, make_fifo_empty())

    return _read


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_state():
    """Reset shared module state and the DMA mock before every test."""
    fc._ack_expected_seq = 0
    _mock_dmadump.dma_dump_read.reset_mock(side_effect=True, return_value=True)
    _mock_dmadump.dma_dump_write.reset_mock()
    _mock_dmadump.dma_dump_write.return_value = 0
    yield


# ===========================================================================
# 1. C2H header constants  (spec §4)
# ===========================================================================

class TestHeaderConstants:
    """Every constant in the C2H quick-reference table must match the spec."""

    def test_ack(self):
        # 0xACE0_ACE0 → byte[63]=0xAC, byte[62]=0xE0
        assert _HDR_ACK == (0xAC, 0xE0)

    def test_spike(self):
        assert _HDR_SPIKE == (0xEE, 0xEE)

    def test_exec_done(self):
        assert _HDR_EXEC_DONE == (0xAB, 0xCD)

    def test_latency(self):
        assert _HDR_LATENCY == (0xBA, 0xBA)

    def test_hbm_acc(self):
        assert _HDR_HBM_ACC == (0xCA, 0xBA)

    def test_hbm_data(self):
        assert _HDR_HBM_DATA == (0xBB, 0xBB)

    def test_mem_pot(self):
        assert _HDR_MEM_POT == (0xCC, 0xCC)

    def test_fifo_empty(self):
        assert _HDR_FIFO_EMPTY == (0xFF, 0xFF)

    def test_status_ok(self):
        assert ACK_STATUS_OK == 0x01

    def test_status_backpressure(self):
        assert ACK_STATUS_BACKPRESSURE == 0x02


# ===========================================================================
# 2. parse_ack_packet  (spec §2)
# ===========================================================================

class TestParseAckPacket:

    def test_cmd_echo_field(self):
        pkt = make_ack(cmd_echo=0x04)
        cmd, _, _ = parse_ack_packet(pkt)
        assert cmd == 0x04

    def test_status_ok(self):
        pkt = make_ack(0x04, status=ACK_STATUS_OK)
        _, _, stat = parse_ack_packet(pkt)
        assert stat == ACK_STATUS_OK

    def test_status_backpressure(self):
        pkt = make_ack(0x08, status=ACK_STATUS_BACKPRESSURE)
        _, _, stat = parse_ack_packet(pkt)
        assert stat == ACK_STATUS_BACKPRESSURE

    def test_seq_num_zero(self):
        pkt = make_ack(0x02, seq_num=0)
        _, seq, _ = parse_ack_packet(pkt)
        assert seq == 0

    def test_seq_num_multi_byte(self):
        """seq_num spans bytes[58:55] — verify all four bytes are decoded."""
        pkt = make_ack(0x02, seq_num=0x01020304)
        _, seq, _ = parse_ack_packet(pkt)
        assert seq == 0x01020304

    def test_seq_num_max(self):
        """Spec: sequence number rolls over at 2^32."""
        pkt = make_ack(0x02, seq_num=0xFFFFFFFF)
        _, seq, _ = parse_ack_packet(pkt)
        assert seq == 0xFFFFFFFF

    @pytest.mark.parametrize("cmd", [0x02, 0x03, 0x04, 0x08])
    def test_all_acked_commands(self, cmd):
        """All four ACK-generating commands are decoded correctly."""
        pkt = make_ack(cmd_echo=cmd, seq_num=1)
        echo, seq, _ = parse_ack_packet(pkt)
        assert echo == cmd
        assert seq == 1


# ===========================================================================
# 3. wait_for_ack  (spec §6 / §7)
# ===========================================================================

class TestWaitForAck:

    def test_returns_on_first_ack(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(make_ack(0x04, seq_num=3))
        cmd, seq, stat = wait_for_ack(0x04)
        assert cmd == 0x04
        assert seq == 3
        assert stat == ACK_STATUS_OK

    def test_skips_fifo_empty_packets(self):
        """FIFO-empty responses must be discarded silently while polling."""
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            make_fifo_empty(),
            make_fifo_empty(),
            make_ack(0x08, seq_num=1),
        )
        _, seq, _ = wait_for_ack(0x08)
        assert seq == 1
        assert _mock_dmadump.dma_dump_read.call_count == 3

    def test_raises_ack_error_on_timeout(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*[make_fifo_empty()] * 10)
        with pytest.raises(ACKError, match="0x04"):
            wait_for_ack(0x04, timeout_reads=5)

    def test_error_message_includes_cmd_hex(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*[make_fifo_empty()] * 10)
        with pytest.raises(ACKError, match="0x08"):
            wait_for_ack(0x08, timeout_reads=3)

    def test_advances_expected_seq(self):
        """_ack_expected_seq must be updated to seq_num + 1."""
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(make_ack(0x04, seq_num=10))
        wait_for_ack(0x04)
        assert fc._ack_expected_seq == 11

    def test_returns_backpressure_status_to_caller(self):
        pkt = make_ack(0x08, seq_num=99, status=ACK_STATUS_BACKPRESSURE)
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(pkt)
        _, seq, stat = wait_for_ack(0x08)
        assert stat == ACK_STATUS_BACKPRESSURE
        assert seq == 99

    def test_warns_on_cmd_echo_mismatch(self, caplog):
        """ACK with wrong cmd_echo logs a warning but still returns (spec §6)."""
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            make_ack(cmd_echo=0x02, seq_num=0)  # expected 0x04
        )
        with caplog.at_level(logging.WARNING):
            cmd, _, _ = wait_for_ack(0x04)
        assert cmd == 0x02
        assert any("mismatch" in r.message.lower() for r in caplog.records)

    def test_warns_on_backpressure_status(self, caplog):
        """Backpressure status (0x02) must be logged as a warning."""
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            make_ack(0x04, status=ACK_STATUS_BACKPRESSURE)
        )
        with caplog.at_level(logging.WARNING):
            wait_for_ack(0x04)
        assert any("backpressure" in r.message.lower() for r in caplog.records)

    def test_warns_on_unexpected_packet_type(self, caplog):
        """An unrecognised packet is logged but polling continues."""
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            make_spike_pkt(),  # unexpected here
            make_ack(0x04),
        )
        with caplog.at_level(logging.WARNING):
            wait_for_ack(0x04)
        assert any("unexpected" in r.message.lower() for r in caplog.records)


# ===========================================================================
# 4. drain_acks
# ===========================================================================

class TestDrainAcks:

    def test_drains_exact_count(self):
        acks = [make_ack(0x02, seq_num=i) for i in range(5)]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*acks)
        drain_acks(5, 0x02)
        # Last ACK had seq_num=4, so expected must now be 5
        assert fc._ack_expected_seq == 5

    def test_zero_is_noop(self):
        """drain_acks(0, …) must not touch the DMA interface."""
        drain_acks(0, 0x04)
        _mock_dmadump.dma_dump_read.assert_not_called()

    def test_raises_if_too_few_acks(self):
        """If fewer ACKs arrive than expected, ACKError must be raised."""
        acks = [make_ack(0x02, seq_num=i) for i in range(2)]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*acks)
        with pytest.raises(ACKError):
            drain_acks(3, 0x02, timeout_reads=2)

    def test_sequential_seq_numbers(self):
        """drain_acks must process ACKs one-by-one and track the sequence."""
        acks = [make_ack(0x04, seq_num=i) for i in range(4)]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*acks)
        drain_acks(4, 0x04)
        assert fc._ack_expected_seq == 4  # seq 3 → expected 4


# ===========================================================================
# 5. write_parameters_simple — CMD_NTWK_PARAM_W (0x04)  (spec §3)
# ===========================================================================

class TestWriteParametersSimple:
    """0x04 must wait for a hardware ACK after every write."""

    def test_waits_for_ack_0x04(self):
        with patch.object(fc, "wait_for_ack") as mock_wait:
            write_parameters_simple(n_outputs=16, n_inputs=8)
        mock_wait.assert_called_once_with(0x04)

    def test_dma_write_before_ack(self):
        """DMA write must happen before the ACK read."""
        order = []
        _mock_dmadump.dma_dump_write.side_effect = lambda *a, **k: order.append("write") or 0
        with patch.object(fc, "wait_for_ack",
                          side_effect=lambda *a, **k: order.append("ack")):
            write_parameters_simple(n_outputs=16, n_inputs=8)
        assert order == ["write", "ack"]

    def test_no_ack_in_simdump_mode(self):
        """simDump=True bypasses hardware — no ACK must be requested."""
        with patch.object(fc, "wait_for_ack") as mock_wait:
            result = write_parameters_simple(n_outputs=16, n_inputs=8, simDump=True)
        mock_wait.assert_not_called()
        assert result is not None  # returns the command list


# ===========================================================================
# 6. write_neuron_type — CMD_NTWK_PARAM_MEM_W (0x08)  (spec §3)
# ===========================================================================

class TestWriteNeuronType:
    """0x08 must wait for a hardware ACK after every write."""

    def test_waits_for_ack_0x08(self):
        with patch.object(fc, "wait_for_ack") as mock_wait:
            write_neuron_type(stopAddr=16, Threshold=10, neuronModel=2, shift=0, leak=63)
        mock_wait.assert_called_once_with(0x08)

    def test_dma_write_before_ack(self):
        order = []
        _mock_dmadump.dma_dump_write.side_effect = lambda *a, **k: order.append("write") or 0
        with patch.object(fc, "wait_for_ack",
                          side_effect=lambda *a, **k: order.append("ack")):
            write_neuron_type(stopAddr=16, Threshold=10, neuronModel=2, shift=0, leak=63)
        assert order == ["write", "ack"]

    def test_no_ack_in_simdump_mode(self):
        with patch.object(fc, "wait_for_ack") as mock_wait:
            result = write_neuron_type(16, 10, 2, 0, 63, simDump=True)
        mock_wait.assert_not_called()
        assert result is not None


# ===========================================================================
# 7. clear — CMD_IEP_RW (0x03) write path  (spec §3)
# ===========================================================================

class TestClear:
    """clear() sends 16 column packets per row; each generates one ACK."""

    def test_drains_16_acks_for_one_row(self):
        """n_internal=16 → 1 row → drain_acks(16, 0x03) called once."""
        with patch.object(fc, "drain_acks") as mock_drain:
            clear(n_internal=16)
        mock_drain.assert_called_once_with(16, 0x03)

    def test_drains_acks_for_each_row(self):
        """n_internal=32 → 2 rows → drain_acks called twice."""
        with patch.object(fc, "drain_acks") as mock_drain:
            clear(n_internal=32)
        assert mock_drain.call_count == 2
        for c in mock_drain.call_args_list:
            assert c.args == (16, 0x03)

    def test_no_ack_drain_in_simdump_mode(self):
        with patch.object(fc, "drain_acks") as mock_drain:
            clear(n_internal=16, simDump=True)
        mock_drain.assert_not_called()


# ===========================================================================
# 8. read / readSelect — ACK packets skipped while waiting for 0xCCCC
# ===========================================================================

class TestRead:
    """
    Spec §5: ACKs are highest TX priority.  After a batch of CMD_IEP_RW
    read requests, all ACKs arrive before the membrane data packets.
    The read loop must discard ACK packets transparently.
    """

    def test_all_acks_then_membrane_data(self):
        # n_internal=16, ng_num=16 → 1 row, 16 column requests, 16 responses
        reads = (
            [make_ack(0x03, seq_num=i) for i in range(16)]   # ACKs first
            + [make_mem_pot_pkt() for _ in range(16)]         # then membrane data
        )
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        result = read(n_internal=16)
        assert len(result) == 16

    def test_interleaved_acks_and_membrane_data(self):
        """ACK, membrane, ACK, membrane, … alternating pattern."""
        interleaved = []
        for i in range(16):
            interleaved.append(make_ack(0x03, seq_num=i))
            interleaved.append(make_mem_pot_pkt())
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*interleaved)
        result = read(n_internal=16)
        assert len(result) == 16

    def test_no_acks_legacy_behaviour(self):
        """Pure membrane packets (no ACKs) must still work correctly."""
        reads = [make_mem_pot_pkt() for _ in range(16)]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        result = read(n_internal=16)
        assert len(result) == 16

    def test_backpressure_ack_is_logged(self, caplog):
        reads = (
            [make_ack(0x03, seq_num=0, status=ACK_STATUS_BACKPRESSURE)]
            + [make_ack(0x03, seq_num=i + 1) for i in range(15)]
            + [make_mem_pot_pkt() for _ in range(16)]
        )
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with caplog.at_level(logging.WARNING):
            read(n_internal=16)
        assert any("backpressure" in r.message.lower() for r in caplog.records)


class TestReadSelect:
    """readSelect uses the same ACK-skipping loop as read(), one neuron at a time."""

    def test_skips_ack_then_returns_membrane(self):
        reads = [make_ack(0x03), make_mem_pot_pkt()]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        result = readSelect(neuronList=[0])
        assert len(result) == 1

    def test_multiple_neurons(self):
        reads = []
        for _ in range(3):
            reads.append(make_ack(0x03))
            reads.append(make_mem_pot_pkt())
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        result = readSelect(neuronList=[0, 1, 2])
        assert len(result) == 3

    def test_no_acks_still_works(self):
        reads = [make_mem_pot_pkt() for _ in range(2)]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        result = readSelect(neuronList=[0, 1])
        assert len(result) == 2


# ===========================================================================
# 9. flush_spikes — normal path, pending ACKs, and tail-packet headers
# ===========================================================================

class TestFlushSpikes:
    """
    Normal execution-done sequence:
      [optional spikes] → ABCD_ABCD → BABA_BABA → CABA_CABA

    Pending ACKs (spec §5): if a configuration command was sent just before
    execution, its ACK may arrive mixed into the spike stream after the FPGA
    returns to IDLE.  flush_spikes must handle it gracefully.
    """

    def _done_tail(self, latency=100, hbm_acc=7):
        return [make_exec_done_pkt(), make_latency_pkt(latency), make_hbm_acc_pkt(hbm_acc)]

    def test_basic_flush_returns_empty_spikes(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*self._done_tail())
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            spikes, latency, hbm_acc = flush_spikes()
        assert spikes == []
        assert latency == 100
        assert hbm_acc == 7

    def test_latency_value_extracted_correctly(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            *self._done_tail(latency=0xDEAD)
        )
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            _, latency, _ = flush_spikes()
        assert latency == 0xDEAD

    def test_hbm_acc_value_extracted_correctly(self):
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(
            *self._done_tail(hbm_acc=0xBEEF)
        )
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            _, _, hbm_acc = flush_spikes()
        assert hbm_acc == 0xBEEF

    def test_spike_packets_before_exec_done(self):
        reads = [make_spike_pkt(), make_spike_pkt()] + self._done_tail()
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            spikes, _, _ = flush_spikes()
        assert isinstance(spikes, list)

    def test_pending_ack_during_flush_handled_gracefully(self, caplog):
        """
        Spec §5: ACKs queued during execution are delivered when the core
        returns to IDLE.  They must not crash flush_spikes.
        """
        reads = [
            make_ack(0x04, seq_num=5),  # pending config ACK
            make_exec_done_pkt(),
            make_latency_pkt(),
            make_hbm_acc_pkt(),
        ]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            with caplog.at_level(logging.WARNING):
                spikes, _, _ = flush_spikes()
        assert isinstance(spikes, list)
        assert any("ack" in r.message.lower() for r in caplog.records)

    def test_warns_on_wrong_latency_header(self, caplog):
        """If BABA_BABA is missing, a warning must be emitted."""
        reads = [
            make_exec_done_pkt(),
            make_hbm_acc_pkt(),   # wrong packet where latency expected
            make_latency_pkt(),
        ]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            with caplog.at_level(logging.WARNING):
                flush_spikes()
        assert any(
            "baba" in r.message.lower() or "latency" in r.message.lower()
            for r in caplog.records
        )

    def test_warns_on_wrong_hbm_acc_header(self, caplog):
        """If CABA_CABA is missing, a warning must be emitted."""
        reads = [
            make_exec_done_pkt(),
            make_latency_pkt(),
            make_latency_pkt(),   # wrong packet where HBM acc expected
        ]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            with caplog.at_level(logging.WARNING):
                flush_spikes()
        assert any(
            "caba" in r.message.lower() or "hbm" in r.message.lower()
            for r in caplog.records
        )

    def test_fifo_empty_loop_terminates(self):
        """
        If no execution-done packet ever arrives, the 50-empty-packet counter
        must break the loop cleanly.
        """
        reads = [make_fifo_empty()] * 60 + [make_latency_pkt(), make_hbm_acc_pkt()]
        _mock_dmadump.dma_dump_read.side_effect = seq_reads(*reads)
        with patch("hs_bridge.FPGA_Execution.fpga_controller.time"):
            spikes, _, _ = flush_spikes()
        assert spikes == []


# ===========================================================================
# 10. ACKError exception class
# ===========================================================================

class TestACKError:
    def test_is_runtime_error(self):
        assert issubclass(ACKError, RuntimeError)

    def test_carries_message(self):
        with pytest.raises(ACKError, match="timeout waiting"):
            raise ACKError("timeout waiting for 0x04")

    def test_importable_from_fpga_controller(self):
        assert hasattr(fc, "ACKError")

    def test_importable_from_hs_bridge(self):
        import hs_bridge.FPGA_Execution.fpga_controller as mod
        assert hasattr(mod, "ACKError")
