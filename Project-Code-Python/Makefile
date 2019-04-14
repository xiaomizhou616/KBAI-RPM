.phony: all debug plot clean

all:
	@env XH_DEBUG=0 python RavensProject.py
	@echo "============= Result Summary ==============="
	@cat SetResults.csv
	@echo "============= Incorrect answers ============"
	@egrep -i '(incorrect|skipped)' ProblemResults.csv

debug:
	@env XH_DEBUG=1 python RavensProject.py
	@echo "============= Result Summary ==============="
	@cat SetResults.csv
	@echo "============= Incorrect answers ============"
	@egrep -i '(incorrect|skipped)' ProblemResults.csv

plot:
	@mkdir -p rms_plot
	@env PYTHONHOME=$(VIRTUAL_ENV) /usr/local/bin/python RavensPlotter.py rms_plot

clean:
	@rm -rf rms_plot