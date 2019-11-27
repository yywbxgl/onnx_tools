#pragma once
#include <unordered_set>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>

// #include "common/interned_strings.h"

//不允许对象直接复制与拷贝
// #define ONNX_DISALLOW_COPY_AND_ASSIGN(TypeName) \
//   TypeName(const TypeName&) = delete; \
//   void operator=(const TypeName&) = delete


namespace ONNX_NAMESPACE {

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
// 使用别名，方便代码修改
// using NodeKind = Symbol;

class Node {


};

class Value {

};


class Graph {
// ONNX_DISALLOW_COPY_AND_ASSIGN(Graph);
public:
    std::string name_;    //graph name
    std::unordered_set<const Node*> all_nodes;   // all node map
    Node * const output_;   //output node
    Node * const input_;    //input node
};

}// namespace ONNX_NAMESPACE
