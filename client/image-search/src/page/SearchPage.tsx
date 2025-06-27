import React, { useEffect } from "react";
import UploadImage from "../component/UploadImage/UploadImage";
import SearchResults from "../component/SearchResults";

const SearchPage: React.FC = () => {
  const [results, setResults] = React.useState<string[]>([]);
  const [isLoading, setLoading] = React.useState(false);

  useEffect(() => {
    console.log("results", results);
    console.log("isLoading", isLoading);
  }, [results]);
  return (
    <div>
      <UploadImage setResults={setResults} setLoading={setLoading} />
      <SearchResults results={results} isLoading={isLoading} />
    </div>
  );
};
export default SearchPage;
