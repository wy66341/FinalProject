PYTHON := $(CURDIR)/.venv/bin/python3
SRC := $(CURDIR)/src

.PHONY: all pdf clean test figures scan verify sensitivity

all: test figures
	@echo "=== make all done ==="

pdf: report.tex
	xelatex report.tex && xelatex report.tex && xelatex report.tex
	@echo "report.pdf generated."

figures:
	@echo "Generating orbit plots..."
	$(PYTHON) $(SRC)/visualize.py
	@echo "Figures done."

test:
	@echo "=== M1: Patched Conic Validation ==="
	$(PYTHON) $(SRC)/conic_patch.py --test
	@echo ""
	@echo "=== M2: Two-body Circular Benchmark ==="
	$(PYTHON) $(SRC)/nbody.py --test
	@echo ""
	@echo "=== All tests passed ==="

scan:
	$(PYTHON) $(SRC)/optimizer.py

verify:
	$(PYTHON) $(SRC)/horizons_verify.py

sensitivity:
	$(PYTHON) $(SRC)/sensitivity.py

clean:
	rm -f report.pdf report.aux report.log report.out report.toc report.synctex.gz
	rm -f data/*.png animation.mp4
	rm -rf src/__pycache__
