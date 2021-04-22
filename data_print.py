from data_prep import (
    names_traina,
    origins_traina,
    Nnames_train,
    Norigins_train,
    Nnames_test,
    Norigins_test,
    a,
    b,
    dicti,
    Forigins_train,
    Forigins_test,
    char_length,
    char,
    Wordmax,
)


def printout():

    print("Sample Data set:")
    for name, origin in zip(names_traina[:20], origins_traina[:20]):
        print(name.ljust(20), origin)

    print("\n")
    print("Data Set's Shape:")
    print(f"Names Train Shape: {Nnames_train.shape}")
    print(f"Origins Train Shape:  {Norigins_train.shape}")
    print(f"Names Test Shape: {Nnames_test.shape}")
    print(f"Origins Test Shape: {Norigins_test.shape}")

    print("\n")
    print("Uniques Values:")
    print("Test Data Unique Values Count: " + str(len(a)))
    print("Train Data Unique Values Count: " + str(len(b)))

    print("\n")
    print("Reverse Lookup Dictionary:")
    print(dicti)

    print("\n")
    print("Origins Shape - with one hot:")
    print("Origins_train one hot shape: " + str(Forigins_train.shape))
    print("Origins_test one hot shape: " + str(Forigins_test.shape))

    print("Character Dictionary List:")
    print("\nNumber of Characters: " + str(char_length))
    print(char)

    print("\nMax Word length:")
    print(Wordmax)