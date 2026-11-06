"""
Root entry point for the Poseidon SBPQ experiment.
"""

from SBPQ.poseidon.run_sbpq_network_global import main
import sys


if __name__ == "__main__":
    main(sys.argv[1:])
