#pragma once

namespace ArrayUtils
{
template <typename T> void fillWithStep(std::vector<T> &array, const T &value, size_t step)
{
    for (size_t i = 0; i < array.size(); i += step)
    {
        array[i] = value;
    }
}
} // namespace ArrayUtils
