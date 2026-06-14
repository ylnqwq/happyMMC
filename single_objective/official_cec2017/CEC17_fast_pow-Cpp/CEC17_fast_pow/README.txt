The code is intended to improve the performance of the CEC 2017 benchmark functions.
The pow() function calls are replaced by multiplication, where possible.
This gives up to 50-60% faster calculation on some functions (F20, F30).
The modified version is called "cec17_test_fast_pow.cpp".

//Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)