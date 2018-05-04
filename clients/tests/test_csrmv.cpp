/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrmv.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, double, base> csrmv_tuple;

int csr_N_range[] = {12000, 15332, 22031};
int csr_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};
std::vector<double> csr_alpha_range = {1.0, 0.0};
base csr_idxBase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_csrmv : public testing::TestWithParam<csrmv_tuple>
{
    protected:
    parameterized_csrmv() {}
    virtual ~parameterized_csrmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmv_arguments(csrmv_tuple tup)
{
    Arguments arg;
    arg.N       = std::get<0>(tup);
    arg.nnz     = std::get<1>(tup);
    arg.alpha   = std::get<2>(tup);
    arg.idxBase = std::get<3>(tup);
    arg.timing  = 0;
    return arg;
}

TEST(csrmv_bad_arg, csrmv_float)
{
    testing_csrmv_bad_arg<rocsparse_int, float>();
}

TEST_P(parameterized_csrmv, csrmv_float)
{
    Arguments arg = setup_csrmv_arguments(GetParam());
    rocsparse_status status = testing_csrmv<rocsparse_int, float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrmv, csrmv_double)
{
    Arguments arg = setup_csrmv_arguments(GetParam());
    rocsparse_status status = testing_csrmv<rocsparse_int, double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrmv, parameterized_csrmv,
                        testing::Combine(testing::ValuesIn(csr_N_range),
                                         testing::ValuesIn(csr_nnz_range),
                                         testing::ValuesIn(csr_alpha_range),
                                         testing::ValuesIn(csr_idxBase_range)));
