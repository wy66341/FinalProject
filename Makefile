.PHONY: all pdf clean test

all: report.pdf

# --- Numerical computation & figures ---
figures: src/conic_patch.py src/nbody.py src/lunar_swingby.py src/trajectory.py src/optimizer.py src/visualize.py
	cd src && python conic_patch.py
	cd src && python nbody.py
	cd src && python lunar_swingby.py
	cd src && python trajectory.py
	cd src && python optimizer.py
	cd src && python visualize.py

# --- Orbit animation ---
animation.mp4: src/animate.py
	cd src && python animate.py

# --- Compile LaTeX report ---
report.pdf: report.tex figures animation.mp4
	xelatex report.tex
	xelatex report.tex
	xelatex report.tex

pdf: report.tex figures
	xelatex report.tex
	xelatex report.tex

# --- Verification tests ---
test:
	cd src && python nbody.py --test
	cd src && python horizons_verify.py

clean:
	rm -f report.pdf report.aux report.log report.out report.toc report.synctex.gz
	rm -f src/*.png src/*.pdf animation.mp4
	rm -rf data/*.json data/*.npz
