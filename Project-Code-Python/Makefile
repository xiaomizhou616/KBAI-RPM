.phony: all debug plot clean profile

all:
	@env XH_DEBUG=0 time python RavensProject.py
	@echo "============= Result Summary ==============="
	@cat SetResults.csv
	@echo "============= Incorrect answers ============"
	@egrep -i '(incorrect|skipped)' ProblemResults.csv

profile:
	@env XH_DEBUG=0 python -m cProfile -s cumtime RavensProject.py 

debug:
	@env XH_DEBUG=1 time python RavensProject.py
	@echo "============= Result Summary ==============="
	@cat SetResults.csv
	@echo "============= Incorrect answers ============"
	@egrep -i '(incorrect|skipped)' ProblemResults.csv

plot:
	@mkdir -p rms_plot
	@env PYTHONHOME=$(VIRTUAL_ENV) /usr/local/bin/python RavensPlotter.py rms_plot

clean:
	@rm -rf rms_plot