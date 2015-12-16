#pragma once

#include <iostream>
#include <fstream>
#include <vector>

template< class T >
void txt2vec(std::vector<T>& v, std::string filename)
{
	std::cout << filename << std::endl;
	std::ifstream file(filename);
	std::string buf;
	while (file && getline(file, buf))
	{
		v.push_back(buf);
	}
}

template< class T >
void vec2txt(std::vector<T>& v, std::string filename)
{
	std::cout << filename << std::endl;
	std::ofstream file(filename);

	for (size_t i = 0; i < v.size(); i++)
	{
		file << v[i] << std::endl;
	}
}