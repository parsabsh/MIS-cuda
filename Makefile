all:
	nvcc src/helpers.cu src/main.cu -o a.out && ./a.out

clean:
	rm -rf a.out