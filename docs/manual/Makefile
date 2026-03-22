# NexusSim Software Manual — Build Makefile
# Usage:
#   make html     — Build HTML documentation
#   make pdf      — Build PDF documentation
#   make all      — Build both HTML and PDF
#   make clean    — Remove all build artifacts

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PDFLATEX      = pdflatex

.PHONY: all html pdf latex clean

all: html pdf

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

latex:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex

pdf: latex
	cd $(BUILDDIR)/latex && $(PDFLATEX) -interaction=nonstopmode NexusSim_Software_Manual.tex > /dev/null 2>&1; true
	cd $(BUILDDIR)/latex && $(PDFLATEX) -interaction=nonstopmode NexusSim_Software_Manual.tex > /dev/null 2>&1; true
	@test -f $(BUILDDIR)/latex/NexusSim_Software_Manual.pdf || (echo "ERROR: PDF not generated" && false)
	cp $(BUILDDIR)/latex/NexusSim_Software_Manual.pdf ../NexusSim_Software_Manual.pdf
	@echo "PDF written to ../NexusSim_Software_Manual.pdf"

clean:
	rm -rf $(BUILDDIR)
