// components/UploadImage/useUploadImage.ts
import { useRef, useState } from "react";
import { searchImageAPI } from "../../api/image.api";

export function useUploadImage() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [image, setImage] = useState<File>(new File([], "empty"));
  const [results, setResults] = useState<string[]>([]);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    const reader = new FileReader();

    reader.onloadend = () => {
      setPreview(reader.result as string);
      setIsLoading(false);
      setImage(file);
    };

    reader.readAsDataURL(file);
  };
  // useEffect(() => {
  //   if (image) {
  //     console.log(image);
  //   }
  // }, [image]);
  const onSearch = async() => {
    console.log(image);
    const formData = new FormData();
    formData.append('image', image);
    console.log(formData);
   
    const res = await searchImageAPI(formData);
    console.log(res.result.map((item: any) => item.image_url));
    setResults(res.result.map((item: any) => item.image_url));
    
  }

  return {
    inputRef,
    handleClick,
    handleChange,
    preview,
    isLoading,
    onSearch,
    results
  };
}
