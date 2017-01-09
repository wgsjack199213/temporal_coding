NAME=main

TARGET=$(NAME)
BIBTEX := bibtex
TGIF   := tgif
XFIG   := xfig
GNUPLOT:= gnuplot

all: $(TARGET).pdf

$(TARGET).pdf: Makefile *.tex *.bib *.fig *.eps *.png
	pdflatex  $(TARGET).tex
	-bibtex --min-crossrefs=100    $(TARGET)
	pdflatex  $(TARGET).tex
	pdflatex  $(TARGET).tex
	pdflatex  $(TARGET).tex

%.pdf : %.fig #Makefile
	fig2dev -L pdf -b 1 $< $@

%.eps : %.dia #Makefile
	dia --nosplash -e $@ $<

%.eps : %.obj
	TMPDIR=/tmp $(TGIF) -print -eps $<

%.pdf : %.eps #Makefile
	epstopdf $<

clean:
	rm -f *.aux */*.log *.log *.out *.bbl *.blg *~ *.bak $(FIGS) $(TARGET).ps $(TARGET).pdf $(TARGET).synctex.gz
