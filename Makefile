PYTHON := $(CURDIR)/.venv/bin/python3
SRC := $(CURDIR)/src
XELATEX := $(HOME)/Library/TinyTeX/bin/universal-darwin/xelatex

.PHONY: all pdf clean test figures scan verify judge judge-dev sensitivity

all: test figures pdf

pdf: report.tex
	@if [ ! -x "$(XELATEX)" ]; then \
		echo "xelatex not found at $(XELATEX)"; \
		echo "Install: brew install --cask basictex"; \
		exit 1; \
	fi
	$(XELATEX) report.tex && $(XELATEX) report.tex && $(XELATEX) report.tex
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

judge:
	@echo "=== 裁判验证：JPL Horizons 接入 + N-体全年对比 ==="
	$(PYTHON) $(SRC)/horizons_verify.py --judge

judge-dev:
	@echo "=== 裁判验证（开发模式——允许解析历表回退）==="
	$(PYTHON) $(SRC)/horizons_verify.py --judge --allow-analytic

sensitivity:
	$(PYTHON) $(SRC)/sensitivity.py

clean:
	rm -f report.pdf report.aux report.log report.out report.toc report.synctex.gz
	rm -f data/*.png animation.mp4
	rm -rf src/__pycache__
