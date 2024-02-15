from lightning_utilities.core.inheritance import get_all_subclasses


def test_get_all_subclasses():
    class A1: ...

    class A2(A1): ...

    class B1: ...

    class B2(B1): ...

    class C(A2, B2): ...

    assert get_all_subclasses(A1) == {A2, C}
    assert get_all_subclasses(A2) == {C}
    assert get_all_subclasses(B1) == {B2, C}
    assert get_all_subclasses(B2) == {C}
    assert get_all_subclasses(C) == set()
