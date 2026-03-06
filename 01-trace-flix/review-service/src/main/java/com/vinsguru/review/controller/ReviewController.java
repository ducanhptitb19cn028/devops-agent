package com.vinsguru.review.controller;

import com.vinsguru.review.dto.ReviewDto;
import com.vinsguru.review.service.ReviewService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
public class ReviewController {

    private static final Logger log = LoggerFactory.getLogger(ReviewController.class);

    private final ReviewService reviewService;

    public ReviewController(ReviewService reviewService) {
        this.reviewService = reviewService;
    }

    @GetMapping("/api/reviews")
    public List<ReviewDto> getReviews(@RequestParam Integer movieId, @RequestHeader Map<String, String> headers){
        log.info("received headers: {}", headers);
        return this.reviewService.getReviews(movieId);
    }

}
