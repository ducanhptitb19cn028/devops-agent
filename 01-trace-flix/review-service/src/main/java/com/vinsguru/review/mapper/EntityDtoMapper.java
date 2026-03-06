package com.vinsguru.review.mapper;

import com.vinsguru.review.dto.ReviewDto;
import com.vinsguru.review.entity.Review;

public class EntityDtoMapper {

    public static ReviewDto toDto(Review review){
        return new ReviewDto(
                review.getId(),
                review.getRating(),
                review.getComment(),
                review.getReviewer()
        );
    }

}
