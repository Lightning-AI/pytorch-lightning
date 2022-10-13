import { Button, Stack, TextField } from "lightning-ui/src/design-system/components";

interface Data {
  [key: string]: string;
}

export function data2dict(data: Data[]) {
  var dict: Data = {};
  for (var i = 0; i < data.length; i++) {
    if (data[i]["name"] === "") {
      continue;
    }
    dict[data[i]["name"]] = data[i]["value"];
  }
  return dict;
}

export default function EnvironmentConfigurator(props: any) {
  const data: Data[] = props.data;
  const setData = props.setData;
  const addItemAllowed = data[data.length - 1].name.length > 0;

  const onItemAdd = () => {
    setData([...data, { name: "", value: "" }]);
  };

  const onItemChange = (fieldName: string, index: number, text: any) => {
    let newData = [...data];

    text = text.trim();
    if (fieldName == "name") {
      text = text.replace(/[^0-9a-zA-Z_]+/gi, "").toUpperCase();
    }

    newData[index][fieldName] = text;
    setData(newData);
  };

  return (
    <Stack spacing={2}>
      {data.map((entry, index) => (
        <Stack direction="row" spacing={1}>
          <TextField
            helperText=""
            onChange={e => onItemChange("name", index, e)}
            placeholder="KEY"
            size="medium"
            statusText=""
            type="text"
            value={entry.name || ""}
          />
          <TextField
            helperText=""
            onChange={e => onItemChange("value", index, e)}
            placeholder="VALUE"
            size="medium"
            statusText=""
            type="text"
            value={entry.value || ""}
          />
        </Stack>
      ))}

      <Button onClick={onItemAdd} text="Add" color="grey" disabled={!addItemAllowed} />
    </Stack>
  );
}
