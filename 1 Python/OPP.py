
# Encapsulation

# class BankAccount:
#     def __init__(self,name,balance):
#         self.name=name
#         self.__balance=balance

#     def get_balance(self):
#         return self.__balance

#     def set_balance(self,newBalance):
#         self.__balance =newBalance

# acc1=BankAccount("Taha",100_000)
# print(acc1.name)
# print(acc1.get_balance())
# print(acc1.set_balance(50_000))
# print(acc1.get_balance())



#  Inheritance

# class Employee:
#     start_time="18pm"
#     end_time="6pm"

# class Teacher(Employee):
#     def __init__(self,subject):
#         self.subject=subject
#         super().__init__()


# t1=Teacher("Math")
# print(t1.subject," ",t1.start_time)



# class teacher:
#     def __init__(self,salary):
#         self.salary=salary

# class Student:
#     def __init__(self,gpa):
#         self.gpa=gpa

# class TA(teacher,Student):
#     def __init__(self, salary,gpa,name):
#         super().__init__(salary)
#         Student.__init__(self,gpa)
#         self.name=name

# ta=TA(20_000,3.5,"Taha")
# print(ta.salary," ",ta.gpa," ",ta.name)


# Abstraction

# from abc import ABC,abstractmethod

# class Animal(ABC):
#     @abstractmethod
#     def make_sound(self):
#         pass

# class Lion(Animal):
#     def make_sound(self):
#         print("Roar")

# l1=Lion()
# l1.make_sound()
        
