class Grandparent(object):

    def my_method(self):
        print("Grandparent")


class Parent(Grandparent):

    def my_method(self):
        print("Parent")


class Child(Parent):

    def my_method(self):
        print("Hello Grandparent")
        super(Parent, self).my_method()


Child().my_method()
