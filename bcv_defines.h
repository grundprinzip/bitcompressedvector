#ifndef BCV_DEFINES_H
#define BCV_DEFINES_H

#define CACHE_LINE_SIZE 64

#define CREATE_MASK(bits, msk) {	\
	msk = 0; \
	switch(bits)\
    {\
        case 1: msk  = CreateMask<1>::mask(); break;\
        case 2: msk  = CreateMask<2>::mask(); break;\
        case 3: msk  = CreateMask<3>::mask(); break;\
        case 4: msk  = CreateMask<4>::mask(); break;\
        case 5: msk  = CreateMask<5>::mask(); break;\
        case 6: msk  = CreateMask<6>::mask(); break;\
        case 7: msk  = CreateMask<7>::mask(); break;\
        case 8: msk  = CreateMask<8>::mask(); break;\
        case 9: msk  = CreateMask<9>::mask(); break;\
        case 10: msk = CreateMask<10>::mask(); break;\
        case 11: msk = CreateMask<11>::mask(); break;\
        case 12: msk = CreateMask<12>::mask(); break;\
        case 13: msk = CreateMask<13>::mask(); break;\
        case 14: msk = CreateMask<14>::mask(); break;\
        case 15: msk = CreateMask<15>::mask(); break;\
        case 16: msk = CreateMask<16>::mask(); break;\
        case 17: msk = CreateMask<17>::mask(); break;\
        case 18: msk = CreateMask<18>::mask(); break;\
        case 19: msk = CreateMask<19>::mask(); break;\
        case 20: msk = CreateMask<20>::mask(); break;\
        case 21: msk = CreateMask<21>::mask(); break;\
        case 22: msk = CreateMask<22>::mask(); break;\
        case 23: msk = CreateMask<23>::mask(); break;\
        case 24: msk = CreateMask<24>::mask(); break;\
        case 25: msk = CreateMask<25>::mask(); break;\
        case 26: msk = CreateMask<26>::mask(); break;\
        case 27: msk = CreateMask<27>::mask(); break;\
        case 28: msk = CreateMask<28>::mask(); break;\
        case 29: msk = CreateMask<29>::mask(); break;\
        case 30: msk = CreateMask<30>::mask(); break;\
        case 31: msk = CreateMask<31>::mask(); break;\
        case 32: msk = CreateMask<32>::mask(); break;\
        case 33: msk = CreateMask<33>::mask(); break;\
        case 34: msk = CreateMask<34>::mask(); break;\
        case 35: msk = CreateMask<35>::mask(); break;\
        case 36: msk = CreateMask<36>::mask(); break;\
        case 37: msk = CreateMask<37>::mask(); break;\
        case 38: msk = CreateMask<38>::mask(); break;\
        case 39: msk = CreateMask<39>::mask(); break;\
        case 40: msk = CreateMask<40>::mask(); break;\
        case 41: msk = CreateMask<41>::mask(); break;\
        case 42: msk = CreateMask<42>::mask(); break;\
        case 43: msk = CreateMask<43>::mask(); break;\
        case 44: msk = CreateMask<44>::mask(); break;\
        case 45: msk = CreateMask<45>::mask(); break;\
        case 46: msk = CreateMask<46>::mask(); break;\
        case 47: msk = CreateMask<47>::mask(); break;\
        case 48: msk = CreateMask<48>::mask(); break;\
        case 49: msk = CreateMask<49>::mask(); break;\
        case 50: msk = CreateMask<50>::mask(); break;\
        case 51: msk = CreateMask<51>::mask(); break;\
        case 52: msk = CreateMask<52>::mask(); break;\
        case 53: msk = CreateMask<53>::mask(); break;\
        case 54: msk = CreateMask<54>::mask(); break;\
        case 55: msk = CreateMask<55>::mask(); break;\
        case 56: msk = CreateMask<56>::mask(); break;\
        case 57: msk = CreateMask<57>::mask(); break;\
        case 58: msk = CreateMask<58>::mask(); break;\
        case 59: msk = CreateMask<59>::mask(); break;\
        case 60: msk = CreateMask<60>::mask(); break;\
        case 61: msk = CreateMask<61>::mask(); break;\
        case 62: msk = CreateMask<62>::mask(); break;\
        case 63: msk = CreateMask<63>::mask(); break;\
        case 64: msk = CreateMask<64>::mask(); break;\
    } \
}

#endif // BCV_DEFINES_H