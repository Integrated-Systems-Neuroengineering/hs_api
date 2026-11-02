v12 hardware test kit (crisdsc0).  Needs /bitstreams/sixteen_core_noc_v12_ila.bit and, for the card-vs-simulation
comparison, round 16b's heavy-spike replay (round16b_out/r_heavy_p0_tx.txt from crisdsc3) in sim/.
  hw_v12.sh   flash; feature word must be 0xff; error_status bits 16-17 (fix 20) must stay 0 in every test group; fast suite (strict spike counts, incl. [10000] [16000] [16384]);
              bio suite; F09 x16; F10 x8; 64-bit long-delay test; isolated firing 64/32-bit (fix 21); plasticity on the card judged by plast_check.py
              (via hw2tx.py); heavy-spike on the card compared step by step with the simulation.  A test that times
              out is recorded as HANG and the card is reflashed before the next; nothing collected = NOT RUN.
Not included: multi-core NoC tests (the NoC compiler integration in hs_bridge is still to be done).
