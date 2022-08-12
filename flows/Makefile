TARGET=../output/merge_flows.pdf

all: $(TARGET)

$(TARGET): merge_flows.tex
	latexmk -f -pdf -output-directory="../output" merge_flows.tex

clean:
	-rm ../output/merge_flows.aux
	-rm ../output/merge_flows.fdb_latexmk
	-rm ../output/merge_flows.fls
	-rm ../output/merge_flows.log

clobber: clean
	-rm ../output/merge_flows.pdf