IMAGE := clip

build:
	docker build -t $(IMAGE) .

shell:
	docker run -it \
		--shm-size=24g \
		-e DISPLAY=:0 \
		-e QT_X11_NO_MITSHM=1 \
		-v $(PWD):/opt/project \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		--rm \
		$(IMAGE) /bin/bash

run-all:
	$(MAKE) build
	$(MAKE) shell -c "cd src && python extract_image_embeddings_preclip.py && python extract_text_embeddings_preclip.py && python compute_preclip_similarity.py && python compute_clip_similarity.py && python evaluate.py"
