package com.example.demo;

import java.util.Comparator;

public class MyComparator  implements Comparator<Integer> {
    @Override
    public int compare(Integer o1, Integer o2) {
//        return o2.compareTo(o1);
        if (o1 == o2) {
            return 0;
        } else if (o1 < o2) {
            return 1;
        } else {
            return -1;
        }
    }
}
