OUTDIR=../output
TARGET=$(OUTDIR)/BirnstielFig1.pdf

all: $(TARGET) Makefile

$(TARGET): BirnstielFig1.tex emission.pdf $(OUTDIR)/imlup.pdf Makefile
	latexmk -f -pdf -output-directory="$(OUTDIR)" BirnstielFig1.tex

clean:
	-rm -r $(TARGET:.pdf=.aux)
	-rm -r $(TARGET:.pdf=.fls)
	-rm -r $(TARGET:.pdf=.log)
	-rm -r $(TARGET:.pdf=.fdb_latexmk)

clobber: clean
	-rm $(TARGET)
