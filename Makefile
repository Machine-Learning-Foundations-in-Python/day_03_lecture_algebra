.PHONY : all latex bibtex view nonstop clean distclean

TARGET=presentation
SOURCE=$(TARGET).tex
BUILD=build
NONSTOP=-interaction nonstopmode -halt-on-error -file-line-error

all:
	pdflatex --shell-escape -output-directory $(BUILD) $(SOURCE) 
	biber 	 -output-directory $(BUILD) $(TARGET)	
	pdflatex --shell-escape -output-directory $(BUILD) $(SOURCE)
	pdflatex --shell-escape -output-directory $(BUILD) $(SOURCE)

latex:
	pdflatex --shell-escape -output-directory $(BUILD) $(SOURCE)
	pdflatex --shell-escape -output-directory $(BUILD) $(SOURCE)

biber:
	biber 	 -output-directory $(BUILD) $(TARGET)

view:
	open ./$(BUILD)/$(TARGET).pdf &

clean:
	rm *.log *.nav *.out *.snm *.toc *.fls *.fdb_latexmk *.aux *.synctex.gz *.bbl *.bcf *.blg *.run.xml
	rm -rf $(BUILD)
	mkdir $(BUILD)
