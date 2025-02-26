all: \
	index.md \
	0_intro.ipynb \
	1_create_server_amd.ipynb \
	1_create_server_nvidia.ipynb \
	2_prepare_data.ipynb \
	3_start_mlflow.ipynb \
	4_mlflow_track_torch.ipynb \
	5_mlflow_track_lightning.ipynb \
	6_mlflow_api.ipynb \
	7_start_ray.ipynb \
	8_submit_ray.ipynb \
	workspace_mlflow/mlflow_api.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_server_amd.ipynb \
	1_create_server_nvidia.ipynb \
	2_prepare_data.ipynb \
	3_start_mlflow.ipynb \
	4_mlflow_track_torch.ipynb \
	5_mlflow_track_lightning.ipynb \
	6_mlflow_api.ipynb \
	7_start_ray.ipynb \
	8_submit_ray.ipynb \
	workspace_mlflow/mlflow_api.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server_options.md \
		snippets/prepare_data.md \
		snippets/start_mlflow.md \
		snippets/mlflow_track_torch.md \
		snippets/mlflow_track_lightning.md \
		snippets/mlflow_api.md \
		snippets/start_ray.md \
		snippets/submit_ray.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_server_amd.ipynb: snippets/create_server_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server_amd.md \
                -o 1_create_server_amd.ipynb  
	sed -i 's/attachment://g' 1_create_server_amd.ipynb

1_create_server_nvidia.ipynb: snippets/create_server_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server_nvidia.md \
                -o 1_create_server_nvidia.ipynb  
	sed -i 's/attachment://g' 1_create_server_nvidia.ipynb

2_prepare_data.ipynb: snippets/prepare_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/prepare_data.md \
                -o 2_prepare_data.ipynb  
	sed -i 's/attachment://g' 2_prepare_data.ipynb


3_start_mlflow.ipynb: snippets/start_mlflow.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/start_mlflow.md \
				-o 3_start_mlflow.ipynb  
	sed -i 's/attachment://g' 3_start_mlflow.ipynb

4_mlflow_track_torch.ipynb: snippets/mlflow_track_torch.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/mlflow_track_torch.md \
				-o 4_mlflow_track_torch.ipynb  
	sed -i 's/attachment://g' 4_mlflow_track_torch.ipynb

5_mlflow_track_lightning.ipynb: snippets/mlflow_track_torch.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/mlflow_track_lightning.md \
				-o 5_mlflow_track_lightning.ipynb  
	sed -i 's/attachment://g' 5_mlflow_track_lightning.ipynb

6_mlflow_api.ipynb: snippets/mlflow_api.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/mlflow_api.md \
				-o 6_mlflow_api.ipynb  
	sed -i 's/attachment://g' 6_mlflow_api.ipynb

workspace_mlflow/mlflow_api.ipynb: snippets/mlflow_api.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/mlflow_api.md \
				-o workspace_mlflow/mlflow_api.ipynb  
	sed -i 's/attachment://g' workspace_mlflow/mlflow_api.ipynb

7_start_ray.ipynb: snippets/start_ray.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/start_ray.md \
				-o 7_start_ray.ipynb  
	sed -i 's/attachment://g' 7_start_ray.ipynb

8_submit_ray.ipynb: snippets/submit_ray.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/submit_ray.md \
				-o 8_submit_ray.ipynb  
	sed -i 's/attachment://g' 8_submit_ray.ipynb
