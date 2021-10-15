#include "kernel_utils.h"

namespace KernelUtils
{
std::string readKernelFile(const std::string &path)
{
    std::ifstream file(path);
    std::string line;
    std::stringstream result;
    while (std::getline(file, line))
    {
        result << line;
        result << '\n';
    }
    return result.str();
}
} // namespace KernelUtils
