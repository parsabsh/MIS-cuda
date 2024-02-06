all:
	nvcc helpers.cu main.cu -o a.out && ./a.out

clean:
	rm -rf a.out