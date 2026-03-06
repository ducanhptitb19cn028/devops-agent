package com.vinsguru.review.dto;

public record ReviewDto(Integer id,
                        Integer rating,
                        String comment,
                        String reviewer) {
}
