package com.vinsguru.movie.client;

import com.vinsguru.movie.dto.ReviewDto;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.client.RestClient;

import java.util.List;

public class ReviewClient {

    private final RestClient restClient;

    public ReviewClient(RestClient restClient) {
        this.restClient = restClient;
    }

    public List<ReviewDto> getReviews(Integer movieId) {
        return this.restClient.get()
                              .uri(b -> b.queryParam("movieId", movieId).build())
                              .retrieve()
                              .body(new ParameterizedTypeReference<List<ReviewDto>>() {
                              });
    }

}
