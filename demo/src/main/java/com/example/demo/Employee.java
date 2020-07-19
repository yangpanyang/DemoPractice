package com.example.demo;

public class Employee implements Comparable {
    String name;
    Integer salary;

    Employee(String name, Integer salary) {
        this.name = name;
        this.salary = salary;
    }

    public String getName () {
        return name;
    }

    public Integer getSalary () {
        return salary;
    }

    @Override
    public String toString() {
        return name + ": " + salary;
    }

    @Override
    public int compareTo(Object o) {
        Employee employee = (Employee)o;
        if (this.salary == employee.salary) {
            return 0;
        } else if (this.salary < employee.salary) {
            return 1;
        } else {
            return -1;
        }
    }
}
