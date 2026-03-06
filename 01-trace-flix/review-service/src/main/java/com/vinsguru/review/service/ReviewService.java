package com.vinsguru.review.service;

import com.vinsguru.review.dto.ReviewDto;
import com.vinsguru.review.mapper.EntityDtoMapper;
import com.vinsguru.review.repository.ReviewRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ReviewService {

    private final ReviewRepository reviewRepository;

    public ReviewService(ReviewRepository reviewRepository) {
        this.reviewRepository = reviewRepository;
    }

    public List<ReviewDto> getReviews(Integer movieId) {
        return this.reviewRepository.findByMovieId(movieId)
                                    .stream()
                                    .map(EntityDtoMapper::toDto)
                                    .toList();
    }

}
