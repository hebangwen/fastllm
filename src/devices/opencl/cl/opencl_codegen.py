#
#  Copyright © 2018, Alibaba Group Holding Limited
#

import os
import sys
major_py_ver = sys.version_info.major

def convert_string_to_hex_list(code_str):
    hex_list = []
    for i in range(len(code_str)):
        hex_ = hex(ord(code_str[i]))
        hex_list.append(hex_)
    return hex_list

def opencl_codegen():
    if len(sys.argv) < 3:
        print("Usage: python opencl_codegen.py cl_kernel_dir output_path")
        exit(0)
    cl_kernel_dir = sys.argv[1]
    output_path = sys.argv[2]
    print("Generating OpenCL Kernels in "+cl_kernel_dir+" to "+output_path)
    if not os.path.exists(cl_kernel_dir):
        print(cl_kernel_dir + " doesn't exist!")

#common.h
    common_header_code = ""
#quantized_common.h
    quantized_common_header_code = ""
#activation_common.h
    activation_common_header_code = ""
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-2:] == ".h" and file_name[:-2] == "quantized_common":
            with open(file_path, "r") as f:
                quantized_common_header_code += f.read()
        elif file_path[-2:] == ".h" and file_name[:-2] == "activation_common":
            with open(file_path, "r") as f:
                activation_common_header_code += f.read()

    opencl_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
                code_str = ""
                for line in f.readlines():
                    if "#include <activation_common.h>" in line:
                        code_str += common_header_code
                        code_str += activation_common_header_code
                    elif "#include <quantized_common.h>" in line:
                        code_str += common_header_code
                        code_str += quantized_common_header_code
                    elif "#include <common.h>" in line:
                        code_str += common_header_code
                    else:
                        code_str += line
                opencl_code_maps[file_name[:-3]] = convert_string_to_hex_list(code_str)

#source model
    opencl_source_map = "#include <map> \n"
    opencl_source_map += "#include <string> \n"
    opencl_source_map += "#include <vector> \n"
    opencl_source_map += "#include \"devices/opencl/opencl_runtime.h\" \n"
    opencl_source_map += "namespace fastllm {\n"
    opencl_source_map += "  std::vector<unsigned char> *OpenCLRuntime::GetOpenCLPrograms(const std::string &name) {\n"
    opencl_source_map += "    static std::map<std::string, std::vector<unsigned char>> OpenCLProgramMap = { \n"

    if major_py_ver == 2:
        items = opencl_code_maps.iteritems()
    else:
        items = opencl_code_maps.items()
    for file_name, file_source in items:
        if file_name[-4:] == "_buf":
            opencl_source_map += "#ifndef MNN_OPENCL_BUFFER_CLOSED\n"
        if file_name[-13:] == "_subgroup_buf":
            opencl_source_map += "#ifdef MNN_SUPPORT_INTEL_SUBGROUP\n"
        opencl_source_map += "    {\n      \""
        opencl_source_map += file_name
        opencl_source_map += "\", \n"
        opencl_source_map += "      { "
        for source_hex in file_source:
            opencl_source_map += source_hex
            opencl_source_map += ","
        opencl_source_map += " } "
        opencl_source_map += "\n    }, \n"
        if file_name[-4:] == "_buf":
            opencl_source_map += "#endif\n"
        if file_name[-13:] == "_subgroup_buf":
            opencl_source_map += "#endif\n"
    opencl_source_map += "    }; \n"
    opencl_source_map += "    if (OpenCLProgramMap.find(name) != OpenCLProgramMap.end()) { \n"
    opencl_source_map += "      return &OpenCLProgramMap[name]; \n"
    opencl_source_map += "    } \n"
    opencl_source_map += "    return nullptr; \n"
    opencl_source_map += "} \n"
    opencl_source_map += "} \n"

    with open(output_path, "w") as w_file:
        w_file.write(opencl_source_map)

    print("Generate OpenCL Source done !!! \n")

if __name__ == '__main__':
    opencl_codegen()
