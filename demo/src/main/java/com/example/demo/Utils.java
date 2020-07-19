package com.example.demo;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;

public class Utils {
    static void test() {
        System.out.println("This is a test function.");
    }

    public static void copyByName(Object src, Object dest) {
        if ((src == null) || (dest == null)) {
            return;
        }
        Map<String, Field> srcFieldMap = getAssignableFields(src);
        Map<String, Field> destFieldMap = getAssignableFields(dest);
        for (String fieldName : srcFieldMap.keySet()) {
            Field srcField = srcFieldMap.get(fieldName);
            if (srcField != null) {
                if (destFieldMap.containsKey(fieldName)) {
                    Field destField = destFieldMap.get(fieldName);
                    if (srcField.getType().equals(destField.getType())) {
                        try {
                            destField.set(dest, srcField.get(src));
                        } catch (Exception e) {
                            System.out.println(e.toString());
                        }
                    }
                }
            }
        }
    }

    private static Map<String, Field> getAssignableFields(Object object) {
        Map<String, Field> map = new HashMap<>();
        if (object != null) {
            for (Field field : object.getClass().getDeclaredFields()) {
                int modifiers = field.getModifiers();
                if ((Modifier.isStatic(modifiers)) || (Modifier.isFinal(modifiers))) {
                    continue;
                }
                field.setAccessible(true);
                map.put(field.getName(), field);
            }
        }
        return map;
    }

    public static <T> T getFirst(T...args) {
        if (args != null) {
            return args[0];
        }
        return null;
    }

    public static <T> void print(T t) {
        if (t instanceof DocumentTemplate) {
            ((DocumentTemplate) t).print();
        }
    }
}
