DEMO = python example.py \
	--num-examples 500 \
	--num-features 5 \
	--feature-cardinality 10 \
	--prior-probability 0.5 \
	--visualization-interval 1 \
	--biased-feature-proportion=0.02 \
	--biased-feature-effect-length 100

ANIMATE =  convert \
	-delay 50 \
	-resize 600000@ \
	-unsharp 0x1 \
	-loop 0 /tmp/adpredictor/*.png  

all:
	test

proto:
	mkdir -p protobufs
	find . -iname '*.proto' | xargs -J % protoc --proto_path=. --python_out=protobufs %
	touch protobufs/__init__.py

freeze:
	pip freeze > requirements.txt

requirements:
	pip install -r requirements.txt

test:
	env/bin/nosetests

demo: | requirements proto
	rm -rf /tmp/adpredictor/*
	$(DEMO) --num-examples 25 --visualization-interval 1
	$(ANIMATE) /tmp/initial_learning.gif

	rm -rf /tmp/adpredictor/*
	$(DEMO) --num-examples 100 --visualization-interval 5
	$(ANIMATE) /tmp/convergence_learning.gif

	rm -rf /tmp/adpredictor/*
	$(DEMO) --num-examples 200 --visualization-interval 10
	$(ANIMATE) /tmp/online_learning.gif

.PHONY:
	test
