sim: data/64.in data/128.in dilated_conv_sim

server: data/128.in dilated_conv_server

dilated_conv_server: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv_server -I ./header

dilated_conv_sim: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv -I ./header --cudart shared -gencode arch=compute_70,code=compute_70

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/64.in: data/generate
	./data/generate 64

data/128.in: data/generate
	./data/generate 128

run_sim: data/128.in dilated_conv_sim
	./dilated_conv_sim -i data/128.in -k data/64.in
	
run_server: data/128.in dilated_conv_server
	./dilated_conv_server data/128.in -k data/64.in

clean:
	rm -f dilated_conv_sim dilated_conv_server
