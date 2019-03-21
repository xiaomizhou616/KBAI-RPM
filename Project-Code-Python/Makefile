.phony: all debug

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

