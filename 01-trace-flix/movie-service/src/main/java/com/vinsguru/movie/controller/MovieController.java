package com.vinsguru.movie.controller;

import com.vinsguru.movie.dto.MovieDto;
import com.vinsguru.movie.service.MovieService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
public class MovieController {

    private static final Logger log = LoggerFactory.getLogger(MovieController.class);

    private final MovieService movieService;

    public MovieController(MovieService movieService) {
        this.movieService = movieService;
    }

    @GetMapping("/api/movies/{movieId}")
    public ResponseEntity<MovieDto> getMovie(@PathVariable Integer movieId, @RequestHeader Map<String, String> headers){
        log.info("received headers: {}", headers);
        return this.movieService.getMovie(movieId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

}
