import os
from tree_sitter import Language, Parser

def load_parser(lib_path="./tree-sitter-python",
                so_path="./tree-sitter-python-language.so",
                language="python"):
    
    # Need to clone the libarary first.
    # For instance, git clone https://github.com/tree-sitter/tree-sitter-python
    
    if not os.path.isfile(so_path):
        Language.build_library(so_path, [lib_path])
    parser = Parser()
    parser.set_language(Language(so_path, language))
    return parser

if __name__ == "__main__":
    load_parser()