#pragma once

#include <random>

namespace ArrayUtils
{
template <typename T> void fillWithRandomValues(std::vector<T> &array)
{
    std::random_device randomDevice;
    std::mt19937 mersenneTwister(randomDevice());
    std::uniform_real_distribution<> distribution(0.0, 1.0);
#pragma omp parallel for default(none), shared(array, distribution, mersenneTwister)
    for (size_t i = 0; i < array.size(); i++)
    {
        array[i] = distribution(mersenneTwister);
    }
}

template <typename T> void makeDiagonallyDominant(std::vector<T> &array, size_t n)
{
#pragma omp parallel for default(none), shared(array, n)
    for (size_t i = 0; i < n; i++)
    {
        array[n * i + i] += static_cast<float>(n);
    }
}

template <typename T>
bool checkEquality(const std::vector<T> &array, const std::vector<T> &referenceArray, T epsilon = 0.0001)
{
    if (array.size() != referenceArray.size())
    {
        return false;
    }
    for (size_t i = 0; i < array.size(); i++)
    {
        if (abs(array[i] - referenceArray[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}
} // namespace ArrayUtils
