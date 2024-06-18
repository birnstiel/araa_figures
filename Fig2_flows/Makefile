OUTDIR=../output
TARGET=$(OUTDIR)/BirnstielFig2.pdf

all: $(TARGET) Makefile

$(TARGET): BirnstielFig2.tex flows.pdf Makefile
	latexmk -f -pdf -output-directory="$(OUTDIR)" BirnstielFig2.tex

clean:
	-rm -r $(TARGET:.pdf=.aux)
	-rm -r $(TARGET:.pdf=.fls)
	-rm -r $(TARGET:.pdf=.log)
	-rm -r $(TARGET:.pdf=.fdb_latexmk)

clobber: clean
	-rm $(TARGET)

