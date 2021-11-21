#pragma once

#include <random>

namespace ArrayUtils
{
template <typename T> void fillWithRandomValues(std::vector<T> &array)
{
    std::random_device randomDevice;
    std::mt19937 mersenneTwister(randomDevice());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);
    for (T &element : array)
    {
        element = distribution(mersenneTwister);
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
