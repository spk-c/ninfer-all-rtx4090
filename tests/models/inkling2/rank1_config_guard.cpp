// Standalone CPU compatibility regression. No CUDA runtime or model loading.
#include "models/inkling2/config.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace ninfer::models::inkling2;
int main(int argc,char** argv) {
    if(argc != 2) return 2;
    std::ifstream in(argv[1]);
    auto cfg = ninfer::artifact::Json::parse(in);
    auto base = parse_text_config(cfg);
    if(base.requires_rank1 || base.visit_count()!=48) return 3;
    cfg["model_type"]="inkling2_rank1_v1";
#ifdef EXPECT_LEGACY
    try { (void)parse_text_config(cfg); }
    catch(const ninfer::artifact::ArtifactError& e) {
        std::cout<<"PASS legacy rejection: "<<e.what()<<"\n"; return 0;
    }
    std::cerr<<"FAIL legacy accepted guarded configuration\n"; return 4;
#else
    auto adapted = parse_text_config(cfg);
    if(!adapted.requires_rank1 || adapted.visit_count()!=base.visit_count()) return 5;
    cfg["model_type"]="inkling2_rank1_v2";
    try { (void)parse_text_config(cfg); }
    catch(const ninfer::artifact::ArtifactError& e) {
        std::cout<<"PASS ordinary and rank1_v1; unknown rank1_v2 rejected: "<<e.what()<<"\n"; return 0;
    }
    return 6;
#endif
}
