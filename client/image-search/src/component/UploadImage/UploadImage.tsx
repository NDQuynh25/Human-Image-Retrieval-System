import React from "react";
import {
  Button,
  Box,
  Typography,
  Paper,
  CircularProgress,
  Stack,
} from "@mui/material";
import UploadIcon from "@mui/icons-material/Upload";
import SearchIcon from "@mui/icons-material/Search";
import { useUploadImage } from "./useUploadImage";

interface UploadImageProps {
  setResults: React.Dispatch<React.SetStateAction<string[]>>;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const UploadImage: React.FC<UploadImageProps> = ({
  setResults,
  setLoading,
}) => {
  const {
    inputRef,
    handleClick,
    handleChange,
    preview,
    isLoading,
    onSearch,
    results,
  } = useUploadImage();

  const [isSearching, setIsSearching] = React.useState(false);
  //const canSearch = !!preview && !isLoading && !isSearching;
  React.useEffect(() => {
    if (results.length > 0) {
      setResults(results);
      setLoading(false);
    }
  }, [results]);
  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        border: "2px dashed #ccc",
        textAlign: "center",
        borderRadius: 2,
        maxWidth: 200,
        mx: "auto",
      }}
    >
      <input
        type="file"
        hidden
        accept="image/*"
        ref={inputRef}
        onChange={handleChange}
      />

      <Stack direction="column" spacing={2}>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={handleClick}
          disabled={isLoading}
        >
          {isLoading ? "Uploading..." : "Upload Image"}
        </Button>

        <Button
          component="button"
          variant="outlined"
          startIcon={<SearchIcon />}
          //disabled={!canSearch}
          onClick={(e) => {
            onSearch();
            setLoading(true);
          }}
        >
          {isSearching ? "Searching..." : "Search"}
        </Button>
      </Stack>

      {isLoading && (
        <Box mt={2}>
          <CircularProgress size={24} />
        </Box>
      )}

      {preview && !isLoading && (
        <Box mt={3}>
          <Typography variant="subtitle2" gutterBottom>
            Preview:
          </Typography>
          <Box
            component="img"
            src={preview}
            alt="preview"
            sx={{
              width: "100%",
              maxWidth: 300,
              borderRadius: 2,
              boxShadow: 1,
            }}
          />
        </Box>
      )}
    </Paper>
  );
};

export default UploadImage;
