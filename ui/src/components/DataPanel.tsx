import React from "react";

type Props = { data: any };

const DataPanel = ({ data }: Props) => {
  return (
    <div className="absolute top-0 right-0 p-4 w-96 h-full z-[1000]">
      <div className="h-full w-full bg-blue-50 p-2 ring rounded-md">
        <p className="font-semibold">Tracking data</p>
        {Object.entries(data.tracking).map(([key, value], index) => {
          return (
            <div>
              <p className="text-xs font-semibold">{key}</p>
              <p>lat {value[0]}</p>
              <p>lon {value[1]}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DataPanel;
