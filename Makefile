GPU ?= amd

FILTER := filters/gpu_select.lua
INDEX_OUT := index.md
INDEX_NVIDIA_OUT := index_nvidia.md
INDEX_AMD_OUT := index_amd.md
CREATE_SERVER_OUT := 2_create_server.ipynb

.PHONY: all build amd nvidia clean validate-gpu

all: build

build: validate-gpu $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) $(INDEX_OUT) 0_intro.ipynb 1_create_lease.ipynb $(CREATE_SERVER_OUT) 3_prepare_data.ipynb 4_start_ray.ipynb 5_submit_ray.ipynb

amd:
	$(MAKE) build GPU=amd

nvidia:
	$(MAKE) build GPU=nvidia

validate-gpu:
	@if [ "$(GPU)" != "nvidia" ] && [ "$(GPU)" != "amd" ]; then \
		echo "Unsupported GPU '$(GPU)'. Use GPU=nvidia or GPU=amd."; \
		exit 1; \
	fi

$(INDEX_NVIDIA_OUT): snippets/intro.md snippets/create_lease.md snippets/create_server_nvidia.md snippets/prepare_data.md snippets/start_ray.md snippets/submit_ray.md snippets/footer.md $(FILTER)
	cat snippets/intro.md \
		snippets/create_lease.md \
		snippets/create_server_nvidia.md \
		snippets/prepare_data.md \
		snippets/start_ray.md \
		snippets/submit_ray.md \
		> index.nvidia.tmp.md
	GPU=nvidia pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.nvidia.filtered.md index.nvidia.tmp.md
	grep -v '^:::' index.nvidia.filtered.md > $(INDEX_NVIDIA_OUT)
	rm index.nvidia.tmp.md index.nvidia.filtered.md
	cat snippets/footer.md >> $(INDEX_NVIDIA_OUT)

$(INDEX_AMD_OUT): snippets/intro.md snippets/create_lease.md snippets/create_server_amd.md snippets/prepare_data.md snippets/start_ray.md snippets/submit_ray.md snippets/footer.md $(FILTER)
	cat snippets/intro.md \
		snippets/create_lease.md \
		snippets/create_server_amd.md \
		snippets/prepare_data.md \
		snippets/start_ray.md \
		snippets/submit_ray.md \
		> index.amd.tmp.md
	GPU=amd pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.amd.filtered.md index.amd.tmp.md
	grep -v '^:::' index.amd.filtered.md > $(INDEX_AMD_OUT)
	rm index.amd.tmp.md index.amd.filtered.md
	cat snippets/footer.md >> $(INDEX_AMD_OUT)

$(INDEX_OUT): $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) validate-gpu
	cp index_$(GPU).md $(INDEX_OUT)

0_intro.ipynb: snippets/intro.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/intro.md \
		-o 0_intro.ipynb
	sed -i 's/attachment://g' 0_intro.ipynb

1_create_lease.ipynb: snippets/create_lease.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/create_lease.md \
		-o 1_create_lease.ipynb
	sed -i 's/attachment://g' 1_create_lease.ipynb

2_create_server.ipynb: snippets/create_server_amd.md snippets/create_server_nvidia.md validate-gpu
	@if [ "$(GPU)" = "amd" ]; then \
		pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
			-i snippets/frontmatter_python.md snippets/create_server_amd.md \
			-o 2_create_server.ipynb; \
	else \
		pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
			-i snippets/frontmatter_python.md snippets/create_server_nvidia.md \
			-o 2_create_server.ipynb; \
	fi
	sed -i 's/attachment://g' 2_create_server.ipynb

3_prepare_data.ipynb: snippets/prepare_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
		-i snippets/frontmatter_python.md snippets/prepare_data.md \
		-o 3_prepare_data.ipynb
	sed -i 's/attachment://g' 3_prepare_data.ipynb

4_start_ray.ipynb: snippets/start_ray.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/start_ray.md \
		-o 4_start_ray.ipynb
	sed -i 's/attachment://g' 4_start_ray.ipynb

5_submit_ray.ipynb: snippets/submit_ray.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
		-i snippets/frontmatter_python.md snippets/submit_ray.md \
		-o 5_submit_ray.ipynb
	sed -i 's/attachment://g' 5_submit_ray.ipynb

clean:
	rm -f index.md index_nvidia.md index_amd.md \
		index.nvidia.tmp.md index.amd.tmp.md index.nvidia.filtered.md index.amd.filtered.md \
		0_intro.ipynb \
		1_create_lease.ipynb \
		2_create_server.ipynb \
		3_prepare_data.ipynb \
		4_start_ray.ipynb \
		5_submit_ray.ipynb
