// components/SearchResults/SearchResults.tsx
import React from "react";
import {
  Box,
  Typography,
  Card,
  CardMedia,
  CardContent,
  CircularProgress,
} from "@mui/material";
import Grid from "@mui/material/Grid";

interface SearchResultsProps {
  results: string[]; // Mảng URL ảnh
  isLoading: boolean;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  isLoading,
}) => {
  if (results.length === 0) {
    return (
      <Box mt={4} textAlign="center">
        <Typography variant="subtitle1" color="text.secondary">
          Chưa có kết quả nào.
        </Typography>
      </Box>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        marginTop: "20px",
        gap: "20px",
      }}
    >
      {isLoading === true ? (
        <CircularProgress />
      ) : (
        <Grid container spacing={2}>
          {results.map((url, index) => (
            <div key={index}>
              <Card elevation={3}>
                <CardMedia
                  component="img"
                  image={url}
                  alt={`result-${index}`}
                  height="350"
                  sx={{ objectFit: "cover" }}
                />
              </Card>
            </div>
          ))}
        </Grid>
      )}
    </div>
  );
};

export default SearchResults;
